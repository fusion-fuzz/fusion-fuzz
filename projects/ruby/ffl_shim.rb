# Loaded by projects/ruby/driver.py with `-r` in front of every seed.
#
# Most seeds are test methods lifted out of test/ruby/test_*.rb (see
# setup.py). Their bodies call test/unit assertions; without test/unit the
# first `assert_equal` is a NoMethodError and nothing after it runs. This
# defines every assertion the corpus uses as a method that *evaluates* its
# arguments and block and checks nothing: the interpreter still executes
# the expressions under test, and a fused program's outcome is decided by
# whether Ruby survives them, not by whether the original expectation
# holds. Defined on Kernel as private methods so they resolve from any
# receiver context (top level, inside a class body, inside a block).
#
# Deliberately not method_missing on Object: a large share of the corpus
# asserts NoMethodError behaviour, which a catch-all would change.
module Kernel
  FFL_ASSERTIONS = %w[
    assert assert_equal assert_not_equal assert_nil assert_not_nil
    assert_same assert_not_same assert_instance_of assert_kind_of
    assert_respond_to assert_not_respond_to assert_match assert_no_match
    assert_not_match assert_empty assert_not_empty assert_include
    assert_not_include assert_predicate assert_not_predicate
    assert_operator assert_not_operator assert_in_delta assert_in_epsilon
    assert_true assert_false assert_send assert_not_send assert_block
    assert_all assert_all? assert_warn assert_warning assert_no_warning
    assert_deprecated_warning assert_performance_warning assert_output
    assert_silent assert_throw assert_syntax_error assert_valid_syntax
    assert_nothing_raised assert_raise assert_raises
    assert_raise_with_message assert_raise_kind_of assert_fixnum
    assert_bignum assert_float assert_float_equal assert_int_equal
    assert_equal_instance assert_equal_not_same assert_frozen
    assert_not_frozen assert_compiles assert_parse assert_prism_eval
    assert_undefined_in assert_locations assert_unicode_age
    assert_time_constructor assert_bytesplice_result assert_redefine_method
    assert_write assert_rindex assert_index assert_byteindex
    assert_byterindex assert_arity assert_encoding assert_strenc
    assert_step assert_file assert_file_not assert_path_exist
    assert_path_not_exist assert_ractor assert_normal_exit
    assert_ruby_status assert_in_out_err assert_separately
    assert_no_memory_leak assert_linear_performance assert_pattern_list
    assert_join_threads assert_cpu_usage_low assert_no_error
    assert_not_ruby_status assert_recovery assert_syntax_error_prism
    assert_valid_syntax_prism assert_stack_overflow assert_warns
    refute refute_equal refute_nil refute_same refute_match refute_empty
    refute_include refute_predicate refute_operator refute_instance_of
    refute_kind_of refute_respond_to refute_in_delta refute_nothing_raised
    omit skip pend flunk pass omit_if omit_unless skip_if skip_unless
    tempfile_with_content
  ]
  # What ran, for the harness's output fingerprint: a script whose
  # assertions are silent prints nothing, so two children that executed
  # different code would otherwise look identical. Names only — the
  # fingerprint masks numbers.
  FFL_SEEN = {}
  FFL_RESCUED = {}
  # A rolling digest of what the assertions actually saw. Names alone
  # were not a fingerprint: four fifths of this corpus calls nothing but
  # `assert_equal`, so every program's summary line read the same and two
  # children that computed completely different values looked identical
  # to the harness's diversity measure (only 36% of state-fused children
  # counted as differing from both parents). Values are folded into a
  # number and only the number is printed — the text of an inspected
  # value would otherwise land in stdout, where a class name such as
  # NoMethodError matches the harness's "this child failed" patterns.
  FFL_STATE = [0, 0]      # [digest, assertions counted]
  FFL_DIGEST_CAP = 2000

  def ffl_fold(text)
    d = FFL_STATE[0]
    text.each_byte { |b| d = (d * 31 + b) & 0xffffffffffff }
    FFL_STATE[0] = d
  end
  private :ffl_fold

  FFL_ASSERTIONS.each do |name|
    define_method(name) do |*args, **kw, &blk|
      FFL_SEEN[name] = true
      if FFL_STATE[1] < FFL_DIGEST_CAP
        FFL_STATE[1] += 1
        begin
          ffl_fold(args.inspect[0, 200]) unless args.empty?
        rescue Exception
          ffl_fold("!")
        end
      end
      if blk
        begin
          v = blk.call
          begin
            ffl_fold(v.inspect[0, 120]) if FFL_STATE[1] < FFL_DIGEST_CAP
          rescue Exception
            ffl_fold("!")
          end
          v
        rescue Exception => e
          FFL_RESCUED[e.class.name || "anon"] = true
          ffl_fold(e.class.name.to_s)
          e
        end
      else
        args
      end
    end
    private name
  end
  at_exit do
    # stdout, not stderr: the harness reads the first stderr line as the
    # failure diagnostic, and this line is not one.
    # Base-20 over the letters g..z, deliberately: the harness masks
    # digits and long hex runs out of a program's output before
    # fingerprinting it, so a decimal or hex digest would all reduce to
    # the same "#" and the fingerprint would carry nothing.
    alphabet = ("g".."z").to_a
    d = FFL_STATE[0]
    enc = d.zero? ? "g" : ""
    while d > 0
      enc = alphabet[d % 20] + enc
      d /= 20
    end
    $stdout.puts "ffl-shim ran=#{FFL_SEEN.keys.sort.join(',')} " \
                 "digest=#{enc} rescued=#{FFL_RESCUED.size}"
  end
  # Bug tracker references (`bug = '[ruby-core:12345]'`) and the
  # `feature` helpers are plain values in the originals.
  def bug(*a) = a
  private :bug
end
